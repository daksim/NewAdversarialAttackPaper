# Latest Adversarial Attack Papers
**update at 2022-10-18 16:59:54**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Marksman Backdoor: Backdoor Attacks with Arbitrary Target Class**

cs.CR

Accepted to NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09194v1) [paper-pdf](http://arxiv.org/pdf/2210.09194v1)

**Authors**: Khoa D. Doan, Yingjie Lao, Ping Li

**Abstract**: In recent years, machine learning models have been shown to be vulnerable to backdoor attacks. Under such attacks, an adversary embeds a stealthy backdoor into the trained model such that the compromised models will behave normally on clean inputs but will misclassify according to the adversary's control on maliciously constructed input with a trigger. While these existing attacks are very effective, the adversary's capability is limited: given an input, these attacks can only cause the model to misclassify toward a single pre-defined or target class. In contrast, this paper exploits a novel backdoor attack with a much more powerful payload, denoted as Marksman, where the adversary can arbitrarily choose which target class the model will misclassify given any input during inference. To achieve this goal, we propose to represent the trigger function as a class-conditional generative model and to inject the backdoor in a constrained optimization framework, where the trigger function learns to generate an optimal trigger pattern to attack any target class at will while simultaneously embedding this generative backdoor into the trained model. Given the learned trigger-generation function, during inference, the adversary can specify an arbitrary backdoor attack target class, and an appropriate trigger causing the model to classify toward this target class is created accordingly. We show empirically that the proposed framework achieves high attack performance while preserving the clean-data performance in several benchmark datasets, including MNIST, CIFAR10, GTSRB, and TinyImageNet. The proposed Marksman backdoor attack can also easily bypass existing backdoor defenses that were originally designed against backdoor attacks with a single target class. Our work takes another significant step toward understanding the extensive risks of backdoor attacks in practice.



## **2. Adversarial Robustness is at Odds with Lazy Training**

cs.CR

NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2207.00411v2) [paper-pdf](http://arxiv.org/pdf/2207.00411v2)

**Authors**: Yunjuan Wang, Enayat Ullah, Poorya Mianjy, Raman Arora

**Abstract**: Recent works show that adversarial examples exist for random neural networks [Daniely and Schacham, 2020] and that these examples can be found using a single step of gradient ascent [Bubeck et al., 2021]. In this work, we extend this line of work to "lazy training" of neural networks -- a dominant model in deep learning theory in which neural networks are provably efficiently learnable. We show that over-parametrized neural networks that are guaranteed to generalize well and enjoy strong computational guarantees remain vulnerable to attacks generated using a single step of gradient ascent.



## **3. DE-CROP: Data-efficient Certified Robustness for Pretrained Classifiers**

cs.LG

WACV 2023. Project page: https://sites.google.com/view/decrop

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08929v1) [paper-pdf](http://arxiv.org/pdf/2210.08929v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Anirban Chakraborty

**Abstract**: Certified defense using randomized smoothing is a popular technique to provide robustness guarantees for deep neural networks against l2 adversarial attacks. Existing works use this technique to provably secure a pretrained non-robust model by training a custom denoiser network on entire training data. However, access to the training set may be restricted to a handful of data samples due to constraints such as high transmission cost and the proprietary nature of the data. Thus, we formulate a novel problem of "how to certify the robustness of pretrained models using only a few training samples". We observe that training the custom denoiser directly using the existing techniques on limited samples yields poor certification. To overcome this, our proposed approach (DE-CROP) generates class-boundary and interpolated samples corresponding to each training sample, ensuring high diversity in the feature space of the pretrained classifier. We train the denoiser by maximizing the similarity between the denoised output of the generated sample and the original training sample in the classifier's logit space. We also perform distribution level matching using domain discriminator and maximum mean discrepancy that yields further benefit. In white box setup, we obtain significant improvements over the baseline on multiple benchmark datasets and also report similar performance under the challenging black box setup.



## **4. Beyond Model Interpretability: On the Faithfulness and Adversarial Robustness of Contrastive Textual Explanations**

cs.CL

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08902v1) [paper-pdf](http://arxiv.org/pdf/2210.08902v1)

**Authors**: Julia El Zini, Mariette Awad

**Abstract**: Contrastive explanation methods go beyond transparency and address the contrastive aspect of explanations. Such explanations are emerging as an attractive option to provide actionable change to scenarios adversely impacted by classifiers' decisions. However, their extension to textual data is under-explored and there is little investigation on their vulnerabilities and limitations.   This work motivates textual counterfactuals by laying the ground for a novel evaluation scheme inspired by the faithfulness of explanations. Accordingly, we extend the computation of three metrics, proximity,connectedness and stability, to textual data and we benchmark two successful contrastive methods, POLYJUICE and MiCE, on our suggested metrics. Experiments on sentiment analysis data show that the connectedness of counterfactuals to their original counterparts is not obvious in both models. More interestingly, the generated contrastive texts are more attainable with POLYJUICE which highlights the significance of latent representations in counterfactual search. Finally, we perform the first semantic adversarial attack on textual recourse methods. The results demonstrate the robustness of POLYJUICE and the role that latent input representations play in robustness and reliability.



## **5. Differential Evolution based Dual Adversarial Camouflage: Fooling Human Eyes and Object Detectors**

cs.CV

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08870v1) [paper-pdf](http://arxiv.org/pdf/2210.08870v1)

**Authors**: Jialiang Sun

**Abstract**: Recent studies reveal that deep neural network (DNN) based object detectors are vulnerable to adversarial attacks in the form of adding the perturbation to the images, leading to the wrong output of object detectors. Most current existing works focus on generating perturbed images, also called adversarial examples, to fool object detectors. Though the generated adversarial examples themselves can remain a certain naturalness, most of them can still be easily observed by human eyes, which limits their further application in the real world. To alleviate this problem, we propose a differential evolution based dual adversarial camouflage (DE_DAC) method, composed of two stages to fool human eyes and object detectors simultaneously. Specifically, we try to obtain the camouflage texture, which can be rendered over the surface of the object. In the first stage, we optimize the global texture to minimize the discrepancy between the rendered object and the scene images, making human eyes difficult to distinguish. In the second stage, we design three loss functions to optimize the local texture, making object detectors ineffective. In addition, we introduce the differential evolution algorithm to search for the near-optimal areas of the object to attack, improving the adversarial performance under certain attack area limitations. Besides, we also study the performance of adaptive DE_DAC, which can be adapted to the environment. Experiments show that our proposed method could obtain a good trade-off between the fooling human eyes and object detectors under multiple specific scenes and objects.



## **6. ODG-Q: Robust Quantization via Online Domain Generalization**

cs.LG

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08701v1) [paper-pdf](http://arxiv.org/pdf/2210.08701v1)

**Authors**: Chaofan Tao, Ngai Wong

**Abstract**: Quantizing neural networks to low-bitwidth is important for model deployment on resource-limited edge hardware. Although a quantized network has a smaller model size and memory footprint, it is fragile to adversarial attacks. However, few methods study the robustness and training efficiency of quantized networks. To this end, we propose a new method by recasting robust quantization as an online domain generalization problem, termed ODG-Q, which generates diverse adversarial data at a low cost during training. ODG-Q consistently outperforms existing works against various adversarial attacks. For example, on CIFAR-10 dataset, ODG-Q achieves 49.2% average improvements under five common white-box attacks and 21.7% average improvements under five common black-box attacks, with a training cost similar to that of natural training (viz. without adversaries). To our best knowledge, this work is the first work that trains both quantized and binary neural networks on ImageNet that consistently improve robustness under different attacks. We also provide a theoretical insight of ODG-Q that accounts for the bound of model risk on attacked data.



## **7. A2: Efficient Automated Attacker for Boosting Adversarial Training**

cs.CV

Accepted by NeurIPS2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.03543v2) [paper-pdf](http://arxiv.org/pdf/2210.03543v2)

**Authors**: Zhuoer Xu, Guanghui Zhu, Changhua Meng, Shiwen Cui, Zhenzhe Ying, Weiqiang Wang, Ming GU, Yihua Huang

**Abstract**: Based on the significant improvement of model robustness by AT (Adversarial Training), various variants have been proposed to further boost the performance. Well-recognized methods have focused on different components of AT (e.g., designing loss functions and leveraging additional unlabeled data). It is generally accepted that stronger perturbations yield more robust models. However, how to generate stronger perturbations efficiently is still missed. In this paper, we propose an efficient automated attacker called A2 to boost AT by generating the optimal perturbations on-the-fly during training. A2 is a parameterized automated attacker to search in the attacker space for the best attacker against the defense model and examples. Extensive experiments across different datasets demonstrate that A2 generates stronger perturbations with low extra cost and reliably improves the robustness of various AT methods against different attacks.



## **8. Reliability and Robustness analysis of Machine Learning based Phishing URL Detectors**

cs.CR

Accepted in Transactions of Dependable and Secure Computing  (SI-Reliability and Robustness in AI-Based Cybersecurity Solutions)

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2005.08454v2) [paper-pdf](http://arxiv.org/pdf/2005.08454v2)

**Authors**: Bushra Sabir, M. Ali Babar, Raj Gaire, Alsharif Abuadbba

**Abstract**: ML-based Phishing URL (MLPU) detectors serve as the first level of defence to protect users and organisations from being victims of phishing attacks. Lately, few studies have launched successful adversarial attacks against specific MLPU detectors raising questions about their practical reliability and usage. Nevertheless, the robustness of these systems has not been extensively investigated. Therefore, the security vulnerabilities of these systems, in general, remain primarily unknown which calls for testing the robustness of these systems. In this article, we have proposed a methodology to investigate the reliability and robustness of 50 representative state-of-the-art MLPU models. Firstly, we have proposed a cost-effective Adversarial URL generator URLBUG that created an Adversarial URL dataset. Subsequently, we reproduced 50 MLPU (traditional ML and Deep learning) systems and recorded their baseline performance. Lastly, we tested the considered MLPU systems on Adversarial Dataset and analyzed their robustness and reliability using box plots and heat maps. Our results showed that the generated adversarial URLs have valid syntax and can be registered at a median annual price of \$11.99. Out of 13\% of the already registered adversarial URLs, 63.94\% were used for malicious purposes. Moreover, the considered MLPU models Matthew Correlation Coefficient (MCC) dropped from a median 0.92 to 0.02 when tested against $Adv_\mathrm{data}$, indicating that the baseline MLPU models are unreliable in their current form. Further, our findings identified several security vulnerabilities of these systems and provided future directions for researchers to design dependable and secure MLPU systems.



## **9. Post-breach Recovery: Protection against White-box Adversarial Examples for Leaked DNN Models**

cs.CR

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2205.10686v2) [paper-pdf](http://arxiv.org/pdf/2205.10686v2)

**Authors**: Shawn Shan, Wenxin Ding, Emily Wenger, Haitao Zheng, Ben Y. Zhao

**Abstract**: Server breaches are an unfortunate reality on today's Internet. In the context of deep neural network (DNN) models, they are particularly harmful, because a leaked model gives an attacker "white-box" access to generate adversarial examples, a threat model that has no practical robust defenses. For practitioners who have invested years and millions into proprietary DNNs, e.g. medical imaging, this seems like an inevitable disaster looming on the horizon.   In this paper, we consider the problem of post-breach recovery for DNN models. We propose Neo, a new system that creates new versions of leaked models, alongside an inference time filter that detects and removes adversarial examples generated on previously leaked models. The classification surfaces of different model versions are slightly offset (by introducing hidden distributions), and Neo detects the overfitting of attacks to the leaked model used in its generation. We show that across a variety of tasks and attack methods, Neo is able to filter out attacks from leaked models with very high accuracy, and provides strong protection (7--10 recoveries) against attackers who repeatedly breach the server. Neo performs well against a variety of strong adaptive attacks, dropping slightly in # of breaches recoverable, and demonstrates potential as a complement to DNN defenses in the wild.



## **10. Robust Feature-Level Adversaries are Interpretability Tools**

cs.LG

Code available at  https://github.com/thestephencasper/feature_level_adv

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2110.03605v5) [paper-pdf](http://arxiv.org/pdf/2110.03605v5)

**Authors**: Stephen Casper, Max Nadeau, Dylan Hadfield-Menell, Gabriel Kreiman

**Abstract**: The literature on adversarial attacks in computer vision typically focuses on pixel-level perturbations. These tend to be very difficult to interpret. Recent work that manipulates the latent representations of image generators to create "feature-level" adversarial perturbations gives us an opportunity to explore perceptible, interpretable adversarial attacks. We make three contributions. First, we observe that feature-level attacks provide useful classes of inputs for studying representations in models. Second, we show that these adversaries are versatile and highly robust. We demonstrate that they can be used to produce targeted, universal, disguised, physically-realizable, and black-box attacks at the ImageNet scale. Third, we show how these adversarial images can be used as a practical interpretability tool for identifying bugs in networks. We use these adversaries to make predictions about spurious associations between features and classes which we then test by designing "copy/paste" attacks in which one natural image is pasted into another to cause a targeted misclassification. Our results suggest that feature-level attacks are a promising approach for rigorous interpretability research. They support the design of tools to better understand what a model has learned and diagnose brittle feature associations. Code is available at https://github.com/thestephencasper/feature_level_adv



## **11. Nowhere to Hide: A Lightweight Unsupervised Detector against Adversarial Examples**

cs.LG

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2210.08579v1) [paper-pdf](http://arxiv.org/pdf/2210.08579v1)

**Authors**: Hui Liu, Bo Zhao, Kehuan Zhang, Peng Liu

**Abstract**: Although deep neural networks (DNNs) have shown impressive performance on many perceptual tasks, they are vulnerable to adversarial examples that are generated by adding slight but maliciously crafted perturbations to benign images. Adversarial detection is an important technique for identifying adversarial examples before they are entered into target DNNs. Previous studies to detect adversarial examples either targeted specific attacks or required expensive computation. How design a lightweight unsupervised detector is still a challenging problem. In this paper, we propose an AutoEncoder-based Adversarial Examples (AEAE) detector, that can guard DNN models by detecting adversarial examples with low computation in an unsupervised manner. The AEAE includes only a shallow autoencoder but plays two roles. First, a well-trained autoencoder has learned the manifold of benign examples. This autoencoder can produce a large reconstruction error for adversarial images with large perturbations, so we can detect significantly perturbed adversarial examples based on the reconstruction error. Second, the autoencoder can filter out the small noise and change the DNN's prediction on adversarial examples with small perturbations. It helps to detect slightly perturbed adversarial examples based on the prediction distance. To cover these two cases, we utilize the reconstruction error and prediction distance from benign images to construct a two-tuple feature set and train an adversarial detector using the isolation forest algorithm. We show empirically that the AEAE is unsupervised and inexpensive against the most state-of-the-art attacks. Through the detection in these two cases, there is nowhere to hide adversarial examples.



## **12. Object-Attentional Untargeted Adversarial Attack**

cs.CV

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2210.08472v1) [paper-pdf](http://arxiv.org/pdf/2210.08472v1)

**Authors**: Chao Zhou, Yuan-Gen Wang, Guopu Zhu

**Abstract**: Deep neural networks are facing severe threats from adversarial attacks. Most existing black-box attacks fool target model by generating either global perturbations or local patches. However, both global perturbations and local patches easily cause annoying visual artifacts in adversarial example. Compared with some smooth regions of an image, the object region generally has more edges and a more complex texture. Thus small perturbations on it will be more imperceptible. On the other hand, the object region is undoubtfully the decisive part of an image to classification tasks. Motivated by these two facts, we propose an object-attentional adversarial attack method for untargeted attack. Specifically, we first generate an object region by intersecting the object detection region from YOLOv4 with the salient object detection (SOD) region from HVPNet. Furthermore, we design an activation strategy to avoid the reaction caused by the incomplete SOD. Then, we perform an adversarial attack only on the detected object region by leveraging Simple Black-box Adversarial Attack (SimBA). To verify the proposed method, we create a unique dataset by extracting all the images containing the object defined by COCO from ImageNet-1K, named COCO-Reduced-ImageNet in this paper. Experimental results on ImageNet-1K and COCO-Reduced-ImageNet show that under various system settings, our method yields the adversarial example with better perceptual quality meanwhile saving the query budget up to 24.16\% compared to the state-of-the-art approaches including SimBA.



## **13. RoS-KD: A Robust Stochastic Knowledge Distillation Approach for Noisy Medical Imaging**

cs.CV

Accepted in ICDM 2022

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08388v1) [paper-pdf](http://arxiv.org/pdf/2210.08388v1)

**Authors**: Ajay Jaiswal, Kumar Ashutosh, Justin F Rousseau, Yifan Peng, Zhangyang Wang, Ying Ding

**Abstract**: AI-powered Medical Imaging has recently achieved enormous attention due to its ability to provide fast-paced healthcare diagnoses. However, it usually suffers from a lack of high-quality datasets due to high annotation cost, inter-observer variability, human annotator error, and errors in computer-generated labels. Deep learning models trained on noisy labelled datasets are sensitive to the noise type and lead to less generalization on the unseen samples. To address this challenge, we propose a Robust Stochastic Knowledge Distillation (RoS-KD) framework which mimics the notion of learning a topic from multiple sources to ensure deterrence in learning noisy information. More specifically, RoS-KD learns a smooth, well-informed, and robust student manifold by distilling knowledge from multiple teachers trained on overlapping subsets of training data. Our extensive experiments on popular medical imaging classification tasks (cardiopulmonary disease and lesion classification) using real-world datasets, show the performance benefit of RoS-KD, its ability to distill knowledge from many popular large networks (ResNet-50, DenseNet-121, MobileNet-V2) in a comparatively small network, and its robustness to adversarial attacks (PGD, FSGM). More specifically, RoS-KD achieves >2% and >4% improvement on F1-score for lesion classification and cardiopulmonary disease classification tasks, respectively, when the underlying student is ResNet-18 against recent competitive knowledge distillation baseline. Additionally, on cardiopulmonary disease classification task, RoS-KD outperforms most of the SOTA baselines by ~1% gain in AUC score.



## **14. GAMA: Generative Adversarial Multi-Object Scene Attacks**

cs.CV

Accepted at NeurIPS 2022; First two authors contributed equally;  Includes Supplementary Material

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2209.09502v2) [paper-pdf](http://arxiv.org/pdf/2209.09502v2)

**Authors**: Abhishek Aich, Calvin-Khang Ta, Akash Gupta, Chengyu Song, Srikanth V. Krishnamurthy, M. Salman Asif, Amit K. Roy-Chowdhury

**Abstract**: The majority of methods for crafting adversarial attacks have focused on scenes with a single dominant object (e.g., images from ImageNet). On the other hand, natural scenes include multiple dominant objects that are semantically related. Thus, it is crucial to explore designing attack strategies that look beyond learning on single-object scenes or attack single-object victim classifiers. Due to their inherent property of strong transferability of perturbations to unknown models, this paper presents the first approach of using generative models for adversarial attacks on multi-object scenes. In order to represent the relationships between different objects in the input scene, we leverage upon the open-sourced pre-trained vision-language model CLIP (Contrastive Language-Image Pre-training), with the motivation to exploit the encoded semantics in the language space along with the visual space. We call this attack approach Generative Adversarial Multi-object scene Attacks (GAMA). GAMA demonstrates the utility of the CLIP model as an attacker's tool to train formidable perturbation generators for multi-object scenes. Using the joint image-text features to train the generator, we show that GAMA can craft potent transferable perturbations in order to fool victim classifiers in various attack settings. For example, GAMA triggers ~16% more misclassification than state-of-the-art generative approaches in black-box settings where both the classifier architecture and data distribution of the attacker are different from the victim. Our code is available here: https://abhishekaich27.github.io/gama.html



## **15. Friendly Noise against Adversarial Noise: A Powerful Defense against Data Poisoning Attacks**

cs.CR

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2208.10224v2) [paper-pdf](http://arxiv.org/pdf/2208.10224v2)

**Authors**: Tian Yu Liu, Yu Yang, Baharan Mirzasoleiman

**Abstract**: A powerful category of (invisible) data poisoning attacks modify a subset of training examples by small adversarial perturbations to change the prediction of certain test-time data. Existing defense mechanisms are not desirable to deploy in practice, as they often either drastically harm the generalization performance, or are attack-specific, and prohibitively slow to apply. Here, we propose a simple but highly effective approach that unlike existing methods breaks various types of invisible poisoning attacks with the slightest drop in the generalization performance. We make the key observation that attacks introduce local sharp regions of high training loss, which when minimized, results in learning the adversarial perturbations and makes the attack successful. To break poisoning attacks, our key idea is to alleviate the sharp loss regions introduced by poisons. To do so, our approach comprises two components: an optimized friendly noise that is generated to maximally perturb examples without degrading the performance, and a randomly varying noise component. The combination of both components builds a very light-weight but extremely effective defense against the most powerful triggerless targeted and hidden-trigger backdoor poisoning attacks, including Gradient Matching, Bulls-eye Polytope, and Sleeper Agent. We show that our friendly noise is transferable to other architectures, and adaptive attacks cannot break our defense due to its random noise component.



## **16. A Scalable Reinforcement Learning Approach for Attack Allocation in Swarm to Swarm Engagement Problems**

cs.RO

submitted to ICRA 2023

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08319v1) [paper-pdf](http://arxiv.org/pdf/2210.08319v1)

**Authors**: Umut Demir, Nazim Kemal Ure

**Abstract**: In this work we propose a reinforcement learning (RL) framework that controls the density of a large-scale swarm for engaging with adversarial swarm attacks. Although there is a significant amount of existing work in applying artificial intelligence methods to swarm control, analysis of interactions between two adversarial swarms is a rather understudied area. Most of the existing work in this subject develop strategies by making hard assumptions regarding the strategy and dynamics of the adversarial swarm. Our main contribution is the formulation of the swarm to swarm engagement problem as a Markov Decision Process and development of RL algorithms that can compute engagement strategies without the knowledge of strategy/dynamics of the adversarial swarm. Simulation results show that the developed framework can handle a wide array of large-scale engagement scenarios in an efficient manner.



## **17. Robust Binary Models by Pruning Randomly-initialized Networks**

cs.LG

Accepted as NeurIPS 2022 paper

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2202.01341v2) [paper-pdf](http://arxiv.org/pdf/2202.01341v2)

**Authors**: Chen Liu, Ziqi Zhao, Sabine Süsstrunk, Mathieu Salzmann

**Abstract**: Robustness to adversarial attacks was shown to require a larger model capacity, and thus a larger memory footprint. In this paper, we introduce an approach to obtain robust yet compact models by pruning randomly-initialized binary networks. Unlike adversarial training, which learns the model parameters, we initialize the model parameters as either +1 or -1, keep them fixed, and find a subnetwork structure that is robust to attacks. Our method confirms the Strong Lottery Ticket Hypothesis in the presence of adversarial attacks, and extends this to binary networks. Furthermore, it yields more compact networks with competitive performance than existing works by 1) adaptively pruning different network layers; 2) exploiting an effective binary initialization scheme; 3) incorporating a last batch normalization layer to improve training stability. Our experiments demonstrate that our approach not only always outperforms the state-of-the-art robust binary networks, but also can achieve accuracy better than full-precision ones on some datasets. Finally, we show the structured patterns of our pruned binary networks.



## **18. Overparameterization from Computational Constraints**

cs.LG

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2208.12926v2) [paper-pdf](http://arxiv.org/pdf/2208.12926v2)

**Authors**: Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Mingyuan Wang

**Abstract**: Overparameterized models with millions of parameters have been hugely successful. In this work, we ask: can the need for large models be, at least in part, due to the \emph{computational} limitations of the learner? Additionally, we ask, is this situation exacerbated for \emph{robust} learning? We show that this indeed could be the case. We show learning tasks for which computationally bounded learners need \emph{significantly more} model parameters than what information-theoretic learners need. Furthermore, we show that even more model parameters could be necessary for robust learning. In particular, for computationally bounded learners, we extend the recent result of Bubeck and Sellke [NeurIPS'2021] which shows that robust models might need more parameters, to the computational regime and show that bounded learners could provably need an even larger number of parameters. Then, we address the following related question: can we hope to remedy the situation for robust computationally bounded learning by restricting \emph{adversaries} to also be computationally bounded for sake of obtaining models with fewer parameters? Here again, we show that this could be possible. Specifically, building on the work of Garg, Jha, Mahloujifar, and Mahmoody [ALT'2020], we demonstrate a learning task that can be learned efficiently and robustly against a computationally bounded attacker, while to be robust against an information-theoretic attacker requires the learner to utilize significantly more parameters.



## **19. Distributionally Robust Multiclass Classification and Applications in Deep Image Classifiers**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2109.12772

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08198v1) [paper-pdf](http://arxiv.org/pdf/2210.08198v1)

**Authors**: Ruidi Chen, Boran Hao, Ioannis Ch. Paschalidis

**Abstract**: We develop a Distributionally Robust Optimization (DRO) formulation for Multiclass Logistic Regression (MLR), which could tolerate data contaminated by outliers. The DRO framework uses a probabilistic ambiguity set defined as a ball of distributions that are close to the empirical distribution of the training set in the sense of the Wasserstein metric. We relax the DRO formulation into a regularized learning problem whose regularizer is a norm of the coefficient matrix. We establish out-of-sample performance guarantees for the solutions to our model, offering insights on the role of the regularizer in controlling the prediction error. We apply the proposed method in rendering deep Vision Transformer (ViT)-based image classifiers robust to random and adversarial attacks. Specifically, using the MNIST and CIFAR-10 datasets, we demonstrate reductions in test error rate by up to 83.5% and loss by up to 91.3% compared with baseline methods, by adopting a novel random training method.



## **20. Dynamics-aware Adversarial Attack of Adaptive Neural Networks**

cs.CV

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08159v1) [paper-pdf](http://arxiv.org/pdf/2210.08159v1)

**Authors**: An Tao, Yueqi Duan, Yingqi Wang, Jiwen Lu, Jie Zhou

**Abstract**: In this paper, we investigate the dynamics-aware adversarial attack problem of adaptive neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed adaptive neural networks, which adaptively deactivate unnecessary execution units based on inputs to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture change afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we reformulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on representative types of adaptive neural networks for both 2D images and 3D point clouds show that our LGM achieves impressive adversarial attack performance compared with the dynamic-unaware attack methods.



## **21. Certified Robustness Against Natural Language Attacks by Causal Intervention**

cs.LG

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2205.12331v3) [paper-pdf](http://arxiv.org/pdf/2205.12331v3)

**Authors**: Haiteng Zhao, Chang Ma, Xinshuai Dong, Anh Tuan Luu, Zhi-Hong Deng, Hanwang Zhang

**Abstract**: Deep learning models have achieved great success in many fields, yet they are vulnerable to adversarial examples. This paper follows a causal perspective to look into the adversarial vulnerability and proposes Causal Intervention by Semantic Smoothing (CISS), a novel framework towards robustness against natural language attacks. Instead of merely fitting observational data, CISS learns causal effects p(y|do(x)) by smoothing in the latent semantic space to make robust predictions, which scales to deep architectures and avoids tedious construction of noise customized for specific attacks. CISS is provably robust against word substitution attacks, as well as empirically robust even when perturbations are strengthened by unknown attack algorithms. For example, on YELP, CISS surpasses the runner-up by 6.7% in terms of certified robustness against word substitutions, and achieves 79.4% empirical robustness when syntactic attacks are integrated.



## **22. SealClub: Computer-aided Paper Document Authentication**

cs.CR

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.07884v1) [paper-pdf](http://arxiv.org/pdf/2210.07884v1)

**Authors**: Martín Ochoa, Jorge Toro-Pozo, David Basin

**Abstract**: Digital authentication is a mature field, offering a range of solutions with rigorous mathematical guarantees. Nevertheless, paper documents, where cryptographic techniques are not directly applicable, are still widely utilized due to usability and legal reasons. We propose a novel approach to authenticating paper documents using smartphones by taking short videos of them. Our solution combines cryptographic and image comparison techniques to detect and highlight subtle semantic-changing attacks on rich documents, containing text and graphics, that could go unnoticed by humans. We rigorously analyze our approach, proving that it is secure against strong adversaries capable of compromising different system components. We also measure its accuracy empirically on a set of 128 videos of paper documents, half containing subtle forgeries. Our algorithm finds all forgeries accurately (no false alarms) after analyzing 5.13 frames on average (corresponding to 1.28 seconds of video). Highlighted regions are large enough to be visible to users, but small enough to precisely locate forgeries. Thus, our approach provides a promising way for users to authenticate paper documents using conventional smartphones under realistic conditions.



## **23. Pre-trained Adversarial Perturbations**

cs.CV

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.03372v2) [paper-pdf](http://arxiv.org/pdf/2210.03372v2)

**Authors**: Yuanhao Ban, Yinpeng Dong

**Abstract**: Self-supervised pre-training has drawn increasing attention in recent years due to its superior performance on numerous downstream tasks after fine-tuning. However, it is well-known that deep learning models lack the robustness to adversarial examples, which can also invoke security issues to pre-trained models, despite being less explored. In this paper, we delve into the robustness of pre-trained models by introducing Pre-trained Adversarial Perturbations (PAPs), which are universal perturbations crafted for the pre-trained models to maintain the effectiveness when attacking fine-tuned ones without any knowledge of the downstream tasks. To this end, we propose a Low-Level Layer Lifting Attack (L4A) method to generate effective PAPs by lifting the neuron activations of low-level layers of the pre-trained models. Equipped with an enhanced noise augmentation strategy, L4A is effective at generating more transferable PAPs against fine-tuned models. Extensive experiments on typical pre-trained vision models and ten downstream tasks demonstrate that our method improves the attack success rate by a large margin compared with state-of-the-art methods.



## **24. Generative Adversarial Learning for Trusted and Secure Clustering in Industrial Wireless Sensor Networks**

cs.NI

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.07707v1) [paper-pdf](http://arxiv.org/pdf/2210.07707v1)

**Authors**: Liu Yang, Simon X. Yang, Yun Li, Yinzhi Lu, Tan Guo

**Abstract**: Traditional machine learning techniques have been widely used to establish the trust management systems. However, the scale of training dataset can significantly affect the security performances of the systems, while it is a great challenge to detect malicious nodes due to the absence of labeled data regarding novel attacks. To address this issue, this paper presents a generative adversarial network (GAN) based trust management mechanism for Industrial Wireless Sensor Networks (IWSNs). First, type-2 fuzzy logic is adopted to evaluate the reputation of sensor nodes while alleviating the uncertainty problem. Then, trust vectors are collected to train a GAN-based codec structure, which is used for further malicious node detection. Moreover, to avoid normal nodes being isolated from the network permanently due to error detections, a GAN-based trust redemption model is constructed to enhance the resilience of trust management. Based on the latest detection results, a trust model update method is developed to adapt to the dynamic industrial environment. The proposed trust management mechanism is finally applied to secure clustering for reliable and real-time data transmission, and simulation results show that it achieves a high detection rate up to 96%, as well as a low false positive rate below 8%.



## **25. Adversarial Pixel Restoration as a Pretext Task for Transferable Perturbations**

cs.CV

Accepted at BMVC'22 (Oral)

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2207.08803v3) [paper-pdf](http://arxiv.org/pdf/2207.08803v3)

**Authors**: Hashmat Shadab Malik, Shahina K Kunhimon, Muzammal Naseer, Salman Khan, Fahad Shahbaz Khan

**Abstract**: Transferable adversarial attacks optimize adversaries from a pretrained surrogate model and known label space to fool the unknown black-box models. Therefore, these attacks are restricted by the availability of an effective surrogate model. In this work, we relax this assumption and propose Adversarial Pixel Restoration as a self-supervised alternative to train an effective surrogate model from scratch under the condition of no labels and few data samples. Our training approach is based on a min-max scheme which reduces overfitting via an adversarial objective and thus optimizes for a more generalizable surrogate model. Our proposed attack is complimentary to the adversarial pixel restoration and is independent of any task specific objective as it can be launched in a self-supervised manner. We successfully demonstrate the adversarial transferability of our approach to Vision Transformers as well as Convolutional Neural Networks for the tasks of classification, object detection, and video segmentation. Our training approach improves the transferability of the baseline unsupervised training method by 16.4% on ImageNet val. set. Our codes & pre-trained surrogate models are available at: https://github.com/HashmatShadab/APR



## **26. When Adversarial Training Meets Vision Transformers: Recipes from Training to Architecture**

cs.CV

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.07540v1) [paper-pdf](http://arxiv.org/pdf/2210.07540v1)

**Authors**: Yichuan Mo, Dongxian Wu, Yifei Wang, Yiwen Guo, Yisen Wang

**Abstract**: Vision Transformers (ViTs) have recently achieved competitive performance in broad vision tasks. Unfortunately, on popular threat models, naturally trained ViTs are shown to provide no more adversarial robustness than convolutional neural networks (CNNs). Adversarial training is still required for ViTs to defend against such adversarial attacks. In this paper, we provide the first and comprehensive study on the adversarial training recipe of ViTs via extensive evaluation of various training techniques across benchmark datasets. We find that pre-training and SGD optimizer are necessary for ViTs' adversarial training. Further considering ViT as a new type of model architecture, we investigate its adversarial robustness from the perspective of its unique architectural components. We find, when randomly masking gradients from some attention blocks or masking perturbations on some patches during adversarial training, the adversarial robustness of ViTs can be remarkably improved, which may potentially open up a line of work to explore the architectural information inside the newly designed models like ViTs. Our code is available at https://github.com/mo666666/When-Adversarial-Training-Meets-Vision-Transformers.



## **27. Characterizing the Influence of Graph Elements**

cs.LG

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.07441v1) [paper-pdf](http://arxiv.org/pdf/2210.07441v1)

**Authors**: Zizhang Chen, Peizhao Li, Hongfu Liu, Pengyu Hong

**Abstract**: Influence function, a method from robust statistics, measures the changes of model parameters or some functions about model parameters concerning the removal or modification of training instances. It is an efficient and useful post-hoc method for studying the interpretability of machine learning models without the need for expensive model re-training. Recently, graph convolution networks (GCNs), which operate on graph data, have attracted a great deal of attention. However, there is no preceding research on the influence functions of GCNs to shed light on the effects of removing training nodes/edges from an input graph. Since the nodes/edges in a graph are interdependent in GCNs, it is challenging to derive influence functions for GCNs. To fill this gap, we started with the simple graph convolution (SGC) model that operates on an attributed graph and formulated an influence function to approximate the changes in model parameters when a node or an edge is removed from an attributed graph. Moreover, we theoretically analyzed the error bound of the estimated influence of removing an edge. We experimentally validated the accuracy and effectiveness of our influence estimation function. In addition, we showed that the influence function of an SGC model could be used to estimate the impact of removing training nodes/edges on the test performance of the SGC without re-training the model. Finally, we demonstrated how to use influence functions to guide the adversarial attacks on GCNs effectively.



## **28. Demystifying Self-supervised Trojan Attacks**

cs.CR

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.07346v1) [paper-pdf](http://arxiv.org/pdf/2210.07346v1)

**Authors**: Changjiang Li, Ren Pang, Zhaohan Xi, Tianyu Du, Shouling Ji, Yuan Yao, Ting Wang

**Abstract**: As an emerging machine learning paradigm, self-supervised learning (SSL) is able to learn high-quality representations for complex data without data labels. Prior work shows that, besides obviating the reliance on labeling, SSL also benefits adversarial robustness by making it more challenging for the adversary to manipulate model prediction. However, whether this robustness benefit generalizes to other types of attacks remains an open question.   We explore this question in the context of trojan attacks by showing that SSL is comparably vulnerable as supervised learning to trojan attacks. Specifically, we design and evaluate CTRL, an extremely simple self-supervised trojan attack. By polluting a tiny fraction of training data (less than 1%) with indistinguishable poisoning samples, CTRL causes any trigger-embedded input to be misclassified to the adversary's desired class with a high probability (over 99%) at inference. More importantly, through the lens of CTRL, we study the mechanisms underlying self-supervised trojan attacks. With both empirical and analytical evidence, we reveal that the representation invariance property of SSL, which benefits adversarial robustness, may also be the very reason making SSL highly vulnerable to trojan attacks. We further discuss the fundamental challenges to defending against self-supervised trojan attacks, pointing to promising directions for future research.



## **29. Autoregressive Perturbations for Data Poisoning**

cs.LG

Accepted to NeurIPS 2022. Code available at  https://github.com/psandovalsegura/autoregressive-poisoning

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2206.03693v3) [paper-pdf](http://arxiv.org/pdf/2206.03693v3)

**Authors**: Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein, David W. Jacobs

**Abstract**: The prevalence of data scraping from social media as a means to obtain datasets has led to growing concerns regarding unauthorized use of data. Data poisoning attacks have been proposed as a bulwark against scraping, as they make data "unlearnable" by adding small, imperceptible perturbations. Unfortunately, existing methods require knowledge of both the target architecture and the complete dataset so that a surrogate network can be trained, the parameters of which are used to generate the attack. In this work, we introduce autoregressive (AR) poisoning, a method that can generate poisoned data without access to the broader dataset. The proposed AR perturbations are generic, can be applied across different datasets, and can poison different architectures. Compared to existing unlearnable methods, our AR poisons are more resistant against common defenses such as adversarial training and strong data augmentations. Our analysis further provides insight into what makes an effective data poison.



## **30. Pikachu: Securing PoS Blockchains from Long-Range Attacks by Checkpointing into Bitcoin PoW using Taproot**

cs.CR

To appear at ConsensusDay 22 (ACM CCS 2022 Workshop)

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2208.05408v2) [paper-pdf](http://arxiv.org/pdf/2208.05408v2)

**Authors**: Sarah Azouvi, Marko Vukolić

**Abstract**: Blockchain systems based on a reusable resource, such as proof-of-stake (PoS), provide weaker security guarantees than those based on proof-of-work. Specifically, they are vulnerable to long-range attacks, where an adversary can corrupt prior participants in order to rewrite the full history of the chain. To prevent this attack on a PoS chain, we propose a protocol that checkpoints the state of the PoS chain to a proof-of-work blockchain such as Bitcoin. Our checkpointing protocol hence does not rely on any central authority. Our work uses Schnorr signatures and leverages Bitcoin recent Taproot upgrade, allowing us to create a checkpointing transaction of constant size. We argue for the security of our protocol and present an open-source implementation that was tested on the Bitcoin testnet.



## **31. AccelAT: A Framework for Accelerating the Adversarial Training of Deep Neural Networks through Accuracy Gradient**

cs.LG

12 pages

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06888v1) [paper-pdf](http://arxiv.org/pdf/2210.06888v1)

**Authors**: Farzad Nikfam, Alberto Marchisio, Maurizio Martina, Muhammad Shafique

**Abstract**: Adversarial training is exploited to develop a robust Deep Neural Network (DNN) model against the malicious altered data. These attacks may have catastrophic effects on DNN models but are indistinguishable for a human being. For example, an external attack can modify an image adding noises invisible for a human eye, but a DNN model misclassified the image. A key objective for developing robust DNN models is to use a learning algorithm that is fast but can also give model that is robust against different types of adversarial attacks. Especially for adversarial training, enormously long training times are needed for obtaining high accuracy under many different types of adversarial samples generated using different adversarial attack techniques.   This paper aims at accelerating the adversarial training to enable fast development of robust DNN models against adversarial attacks. The general method for improving the training performance is the hyperparameters fine-tuning, where the learning rate is one of the most crucial hyperparameters. By modifying its shape (the value over time) and value during the training, we can obtain a model robust to adversarial attacks faster than standard training.   First, we conduct experiments on two different datasets (CIFAR10, CIFAR100), exploring various techniques. Then, this analysis is leveraged to develop a novel fast training methodology, AccelAT, which automatically adjusts the learning rate for different epochs based on the accuracy gradient. The experiments show comparable results with the related works, and in several experiments, the adversarial training of DNNs using our AccelAT framework is conducted up to 2 times faster than the existing techniques. Thus, our findings boost the speed of adversarial training in an era in which security and performance are fundamental optimization objectives in DNN-based applications.



## **32. Adv-Attribute: Inconspicuous and Transferable Adversarial Attack on Face Recognition**

cs.CV

Accepted by NeurIPS2022

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06871v1) [paper-pdf](http://arxiv.org/pdf/2210.06871v1)

**Authors**: Shuai Jia, Bangjie Yin, Taiping Yao, Shouhong Ding, Chunhua Shen, Xiaokang Yang, Chao Ma

**Abstract**: Deep learning models have shown their vulnerability when dealing with adversarial attacks. Existing attacks almost perform on low-level instances, such as pixels and super-pixels, and rarely exploit semantic clues. For face recognition attacks, existing methods typically generate the l_p-norm perturbations on pixels, however, resulting in low attack transferability and high vulnerability to denoising defense models. In this work, instead of performing perturbations on the low-level pixels, we propose to generate attacks through perturbing on the high-level semantics to improve attack transferability. Specifically, a unified flexible framework, Adversarial Attributes (Adv-Attribute), is designed to generate inconspicuous and transferable attacks on face recognition, which crafts the adversarial noise and adds it into different attributes based on the guidance of the difference in face recognition features from the target. Moreover, the importance-aware attribute selection and the multi-objective optimization strategy are introduced to further ensure the balance of stealthiness and attacking strength. Extensive experiments on the FFHQ and CelebA-HQ datasets show that the proposed Adv-Attribute method achieves the state-of-the-art attacking success rates while maintaining better visual effects against recent attack methods.



## **33. Federated Learning for Tabular Data: Exploring Potential Risk to Privacy**

cs.CR

In the proceedings of The 33rd IEEE International Symposium on  Software Reliability Engineering (ISSRE), November 2022

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06856v1) [paper-pdf](http://arxiv.org/pdf/2210.06856v1)

**Authors**: Han Wu, Zilong Zhao, Lydia Y. Chen, Aad van Moorsel

**Abstract**: Federated Learning (FL) has emerged as a potentially powerful privacy-preserving machine learning methodology, since it avoids exchanging data between participants, but instead exchanges model parameters. FL has traditionally been applied to image, voice and similar data, but recently it has started to draw attention from domains including financial services where the data is predominantly tabular. However, the work on tabular data has not yet considered potential attacks, in particular attacks using Generative Adversarial Networks (GANs), which have been successfully applied to FL for non-tabular data. This paper is the first to explore leakage of private data in Federated Learning systems that process tabular data. We design a Generative Adversarial Networks (GANs)-based attack model which can be deployed on a malicious client to reconstruct data and its properties from other participants. As a side-effect of considering tabular data, we are able to statistically assess the efficacy of the attack (without relying on human observation such as done for FL for images). We implement our attack model in a recently developed generic FL software framework for tabular data processing. The experimental results demonstrate the effectiveness of the proposed attack model, thus suggesting that further research is required to counter GAN-based privacy attacks.



## **34. Observed Adversaries in Deep Reinforcement Learning**

cs.LG

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06787v1) [paper-pdf](http://arxiv.org/pdf/2210.06787v1)

**Authors**: Eugene Lim, Harold Soh

**Abstract**: In this work, we point out the problem of observed adversaries for deep policies. Specifically, recent work has shown that deep reinforcement learning is susceptible to adversarial attacks where an observed adversary acts under environmental constraints to invoke natural but adversarial observations. This setting is particularly relevant for HRI since HRI-related robots are expected to perform their tasks around and with other agents. In this work, we demonstrate that this effect persists even with low-dimensional observations. We further show that these adversarial attacks transfer across victims, which potentially allows malicious attackers to train an adversary without access to the target victim.



## **35. COLLIDER: A Robust Training Framework for Backdoor Data**

cs.LG

Accepted to the 16th Asian Conference on Computer Vision (ACCV 2022)

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06704v1) [paper-pdf](http://arxiv.org/pdf/2210.06704v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Deep neural network (DNN) classifiers are vulnerable to backdoor attacks. An adversary poisons some of the training data in such attacks by installing a trigger. The goal is to make the trained DNN output the attacker's desired class whenever the trigger is activated while performing as usual for clean data. Various approaches have recently been proposed to detect malicious backdoored DNNs. However, a robust, end-to-end training approach, like adversarial training, is yet to be discovered for backdoor poisoned data. In this paper, we take the first step toward such methods by developing a robust training framework, COLLIDER, that selects the most prominent samples by exploiting the underlying geometric structures of the data. Specifically, we effectively filter out candidate poisoned data at each training epoch by solving a geometrical coreset selection objective. We first argue how clean data samples exhibit (1) gradients similar to the clean majority of data and (2) low local intrinsic dimensionality (LID). Based on these criteria, we define a novel coreset selection objective to find such samples, which are used for training a DNN. We show the effectiveness of the proposed method for robust training of DNNs on various poisoned datasets, reducing the backdoor success rate significantly.



## **36. A Game Theoretical vulnerability analysis of Adversarial Attack**

cs.GT

Accepted in 17th International Symposium on Visual Computing,2022

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06670v1) [paper-pdf](http://arxiv.org/pdf/2210.06670v1)

**Authors**: Khondker Fariha Hossain, Alireza Tavakkoli, Shamik Sengupta

**Abstract**: In recent times deep learning has been widely used for automating various security tasks in Cyber Domains. However, adversaries manipulate data in many situations and diminish the deployed deep learning model's accuracy. One notable example is fooling CAPTCHA data to access the CAPTCHA-based Classifier leading to the critical system being vulnerable to cybersecurity attacks. To alleviate this, we propose a computational framework of game theory to analyze the CAPTCHA-based Classifier's vulnerability, strategy, and outcomes by forming a simultaneous two-player game. We apply the Fast Gradient Symbol Method (FGSM) and One Pixel Attack on CAPTCHA Data to imitate real-life scenarios of possible cyber-attack. Subsequently, to interpret this scenario from a Game theoretical perspective, we represent the interaction in the Stackelberg Game in Kuhn tree to study players' possible behaviors and actions by applying our Classifier's actual predicted values. Thus, we interpret potential attacks in deep learning applications while representing viable defense strategies in the game theory prospect.



## **37. Understanding Impacts of Task Similarity on Backdoor Attack and Detection**

cs.CR

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.06509v1) [paper-pdf](http://arxiv.org/pdf/2210.06509v1)

**Authors**: Di Tang, Rui Zhu, XiaoFeng Wang, Haixu Tang, Yi Chen

**Abstract**: With extensive studies on backdoor attack and detection, still fundamental questions are left unanswered regarding the limits in the adversary's capability to attack and the defender's capability to detect. We believe that answers to these questions can be found through an in-depth understanding of the relations between the primary task that a benign model is supposed to accomplish and the backdoor task that a backdoored model actually performs. For this purpose, we leverage similarity metrics in multi-task learning to formally define the backdoor distance (similarity) between the primary task and the backdoor task, and analyze existing stealthy backdoor attacks, revealing that most of them fail to effectively reduce the backdoor distance and even for those that do, still much room is left to further improve their stealthiness. So we further design a new method, called TSA attack, to automatically generate a backdoor model under a given distance constraint, and demonstrate that our new attack indeed outperforms existing attacks, making a step closer to understanding the attacker's limits. Most importantly, we provide both theoretic results and experimental evidence on various datasets for the positive correlation between the backdoor distance and backdoor detectability, demonstrating that indeed our task similarity analysis help us better understand backdoor risks and has the potential to identify more effective mitigations.



## **38. On Attacking Out-Domain Uncertainty Estimation in Deep Neural Networks**

cs.LG

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.02191v2) [paper-pdf](http://arxiv.org/pdf/2210.02191v2)

**Authors**: Huimin Zeng, Zhenrui Yue, Yang Zhang, Ziyi Kou, Lanyu Shang, Dong Wang

**Abstract**: In many applications with real-world consequences, it is crucial to develop reliable uncertainty estimation for the predictions made by the AI decision systems. Targeting at the goal of estimating uncertainty, various deep neural network (DNN) based uncertainty estimation algorithms have been proposed. However, the robustness of the uncertainty returned by these algorithms has not been systematically explored. In this work, to raise the awareness of the research community on robust uncertainty estimation, we show that state-of-the-art uncertainty estimation algorithms could fail catastrophically under our proposed adversarial attack despite their impressive performance on uncertainty estimation. In particular, we aim at attacking the out-domain uncertainty estimation: under our attack, the uncertainty model would be fooled to make high-confident predictions for the out-domain data, which they originally would have rejected. Extensive experimental results on various benchmark image datasets show that the uncertainty estimated by state-of-the-art methods could be easily corrupted by our attack.



## **39. On Optimal Learning Under Targeted Data Poisoning**

cs.LG

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.02713v2) [paper-pdf](http://arxiv.org/pdf/2210.02713v2)

**Authors**: Steve Hanneke, Amin Karbasi, Mohammad Mahmoody, Idan Mehalel, Shay Moran

**Abstract**: Consider the task of learning a hypothesis class $\mathcal{H}$ in the presence of an adversary that can replace up to an $\eta$ fraction of the examples in the training set with arbitrary adversarial examples. The adversary aims to fail the learner on a particular target test point $x$ which is known to the adversary but not to the learner. In this work we aim to characterize the smallest achievable error $\epsilon=\epsilon(\eta)$ by the learner in the presence of such an adversary in both realizable and agnostic settings. We fully achieve this in the realizable setting, proving that $\epsilon=\Theta(\mathtt{VC}(\mathcal{H})\cdot \eta)$, where $\mathtt{VC}(\mathcal{H})$ is the VC dimension of $\mathcal{H}$. Remarkably, we show that the upper bound can be attained by a deterministic learner. In the agnostic setting we reveal a more elaborate landscape: we devise a deterministic learner with a multiplicative regret guarantee of $\epsilon \leq C\cdot\mathtt{OPT} + O(\mathtt{VC}(\mathcal{H})\cdot \eta)$, where $C > 1$ is a universal numerical constant. We complement this by showing that for any deterministic learner there is an attack which worsens its error to at least $2\cdot \mathtt{OPT}$. This implies that a multiplicative deterioration in the regret is unavoidable in this case. Finally, the algorithms we develop for achieving the optimal rates are inherently improper. Nevertheless, we show that for a variety of natural concept classes, such as linear classifiers, it is possible to retain the dependence $\epsilon=\Theta_{\mathcal{H}}(\eta)$ by a proper algorithm in the realizable setting. Here $\Theta_{\mathcal{H}}$ conceals a polynomial dependence on $\mathtt{VC}(\mathcal{H})$.



## **40. Alleviating Adversarial Attacks on Variational Autoencoders with MCMC**

cs.LG

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2203.09940v2) [paper-pdf](http://arxiv.org/pdf/2203.09940v2)

**Authors**: Anna Kuzina, Max Welling, Jakub M. Tomczak

**Abstract**: Variational autoencoders (VAEs) are latent variable models that can generate complex objects and provide meaningful latent representations. Moreover, they could be further used in downstream tasks such as classification. As previous work has shown, one can easily fool VAEs to produce unexpected latent representations and reconstructions for a visually slightly modified input. Here, we examine several objective functions for adversarial attack construction proposed previously and present a solution to alleviate the effect of these attacks. Our method utilizes the Markov Chain Monte Carlo (MCMC) technique in the inference step that we motivate with a theoretical analysis. Thus, we do not incorporate any extra costs during training, and the performance on non-attacked inputs is not decreased. We validate our approach on a variety of datasets (MNIST, Fashion MNIST, Color MNIST, CelebA) and VAE configurations ($\beta$-VAE, NVAE, $\beta$-TCVAE), and show that our approach consistently improves the model robustness to adversarial attacks.



## **41. Visual Prompting for Adversarial Robustness**

cs.CV

6 pages, 4 figures, 3 tables

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.06284v1) [paper-pdf](http://arxiv.org/pdf/2210.06284v1)

**Authors**: Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu

**Abstract**: In this work, we leverage visual prompting (VP) to improve adversarial robustness of a fixed, pre-trained model at testing time. Compared to conventional adversarial defenses, VP allows us to design universal (i.e., data-agnostic) input prompting templates, which have plug-and-play capabilities at testing time to achieve desired model performance without introducing much computation overhead. Although VP has been successfully applied to improving model generalization, it remains elusive whether and how it can be used to defend against adversarial attacks. We investigate this problem and show that the vanilla VP approach is not effective in adversarial defense since a universal input prompt lacks the capacity for robust learning against sample-specific adversarial perturbations. To circumvent it, we propose a new VP method, termed Class-wise Adversarial Visual Prompting (C-AVP), to generate class-wise visual prompts so as to not only leverage the strengths of ensemble prompts but also optimize their interrelations to improve model robustness. Our experiments show that C-AVP outperforms the conventional VP method, with 2.1X standard accuracy gain and 2X robust accuracy gain. Compared to classical test-time defenses, C-AVP also yields a 42X inference time speedup.



## **42. A Characterization of Semi-Supervised Adversarially-Robust PAC Learnability**

cs.LG

NeurIPS 2022 camera-ready

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2202.05420v2) [paper-pdf](http://arxiv.org/pdf/2202.05420v2)

**Authors**: Idan Attias, Steve Hanneke, Yishay Mansour

**Abstract**: We study the problem of learning an adversarially robust predictor to test time attacks in the semi-supervised PAC model. We address the question of how many labeled and unlabeled examples are required to ensure learning. We show that having enough unlabeled data (the size of a labeled sample that a fully-supervised method would require), the labeled sample complexity can be arbitrarily smaller compared to previous works, and is sharply characterized by a different complexity measure. We prove nearly matching upper and lower bounds on this sample complexity. This shows that there is a significant benefit in semi-supervised robust learning even in the worst-case distribution-free model, and establishes a gap between the supervised and semi-supervised label complexities which is known not to hold in standard non-robust PAC learning.



## **43. Double Bubble, Toil and Trouble: Enhancing Certified Robustness through Transitivity**

cs.LG

Accepted for Neurips`22, 19 pages, 14 figures, for associated code  see https://github.com/andrew-cullen/DoubleBubble

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.06077v1) [paper-pdf](http://arxiv.org/pdf/2210.06077v1)

**Authors**: Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: In response to subtle adversarial examples flipping classifications of neural network models, recent research has promoted certified robustness as a solution. There, invariance of predictions to all norm-bounded attacks is achieved through randomised smoothing of network inputs. Today's state-of-the-art certifications make optimal use of the class output scores at the input instance under test: no better radius of certification (under the $L_2$ norm) is possible given only these score. However, it is an open question as to whether such lower bounds can be improved using local information around the instance under test. In this work, we demonstrate how today's "optimal" certificates can be improved by exploiting both the transitivity of certifications, and the geometry of the input space, giving rise to what we term Geometrically-Informed Certified Robustness. By considering the smallest distance to points on the boundary of a set of certifications this approach improves certifications for more than $80\%$ of Tiny-Imagenet instances, yielding an on average $5 \%$ increase in the associated certification. When incorporating training time processes that enhance the certified radius, our technique shows even more promising results, with a uniform $4$ percentage point increase in the achieved certified radius.



## **44. SA: Sliding attack for synthetic speech detection with resistance to clipping and self-splicing**

cs.SD

Updated description and formula

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2208.13066v2) [paper-pdf](http://arxiv.org/pdf/2208.13066v2)

**Authors**: Deng JiaCheng, Dong Li, Yan Diqun, Wang Rangding, Zeng Jiaming

**Abstract**: Deep neural networks are vulnerable to adversarial examples that mislead models with imperceptible perturbations. In audio, although adversarial examples have achieved incredible attack success rates on white-box settings and black-box settings, most existing adversarial attacks are constrained by the input length. A More practical scenario is that the adversarial examples must be clipped or self-spliced and input into the black-box model. Therefore, it is necessary to explore how to improve transferability in different input length settings. In this paper, we take the synthetic speech detection task as an example and consider two representative SOTA models. We observe that the gradients of fragments with the same sample value are similar in different models via analyzing the gradients obtained by feeding samples into the model after cropping or self-splicing. Inspired by the above observation, we propose a new adversarial attack method termed sliding attack. Specifically, we make each sampling point aware of gradients at different locations, which can simulate the situation where adversarial examples are input to black-box models with varying input lengths. Therefore, instead of using the current gradient directly in each iteration of the gradient calculation, we go through the following three steps. First, we extract subsegments of different lengths using sliding windows. We then augment the subsegments with data from the adjacent domains. Finally, we feed the sub-segments into different models to obtain aggregate gradients to update adversarial examples. Empirical results demonstrate that our method could significantly improve the transferability of adversarial examples after clipping or self-splicing. Besides, our method could also enhance the transferability between models based on different features.



## **45. Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation**

cs.CV

NeurIPS 2022 conference paper

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.05968v1) [paper-pdf](http://arxiv.org/pdf/2210.05968v1)

**Authors**: Zeyu Qin, Yanbo Fan, Yi Liu, Li Shen, Yong Zhang, Jue Wang, Baoyuan Wu

**Abstract**: Deep neural networks (DNNs) have been shown to be vulnerable to adversarial examples, which can produce erroneous predictions by injecting imperceptible perturbations. In this work, we study the transferability of adversarial examples, which is significant due to its threat to real-world applications where model architecture or parameters are usually unknown. Many existing works reveal that the adversarial examples are likely to overfit the surrogate model that they are generated from, limiting its transfer attack performance against different target models. To mitigate the overfitting of the surrogate model, we propose a novel attack method, dubbed reverse adversarial perturbation (RAP). Specifically, instead of minimizing the loss of a single adversarial point, we advocate seeking adversarial example located at a region with unified low loss value, by injecting the worst-case perturbation (the reverse adversarial perturbation) for each step of the optimization procedure. The adversarial attack with RAP is formulated as a min-max bi-level optimization problem. By integrating RAP into the iterative process for attacks, our method can find more stable adversarial examples which are less sensitive to the changes of decision boundary, mitigating the overfitting of the surrogate model. Comprehensive experimental comparisons demonstrate that RAP can significantly boost adversarial transferability. Furthermore, RAP can be naturally combined with many existing black-box attack techniques, to further boost the transferability. When attacking a real-world image recognition system, Google Cloud Vision API, we obtain 22% performance improvement of targeted attacks over the compared method. Our codes are available at https://github.com/SCLBD/Transfer_attack_RAP.



## **46. Robust Models are less Over-Confident**

cs.CV

accepted at NeuRips 2022

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.05938v1) [paper-pdf](http://arxiv.org/pdf/2210.05938v1)

**Authors**: Julia Grabinski, Paul Gavrikov, Janis Keuper, Margret Keuper

**Abstract**: Despite the success of convolutional neural networks (CNNs) in many academic benchmarks for computer vision tasks, their application in the real-world is still facing fundamental challenges. One of these open problems is the inherent lack of robustness, unveiled by the striking effectiveness of adversarial attacks. Current attack methods are able to manipulate the network's prediction by adding specific but small amounts of noise to the input. In turn, adversarial training (AT) aims to achieve robustness against such attacks and ideally a better model generalization ability by including adversarial samples in the trainingset. However, an in-depth analysis of the resulting robust models beyond adversarial robustness is still pending. In this paper, we empirically analyze a variety of adversarially trained models that achieve high robust accuracies when facing state-of-the-art attacks and we show that AT has an interesting side-effect: it leads to models that are significantly less overconfident with their decisions, even on clean data than non-robust models. Further, our analysis of robust models shows that not only AT but also the model's building blocks (like activation functions and pooling) have a strong influence on the models' prediction confidences. Data & Project website: https://github.com/GeJulia/robustness_confidences_evaluation



## **47. Efficient Adversarial Training without Attacking: Worst-Case-Aware Robust Reinforcement Learning**

cs.LG

36th Conference on Neural Information Processing Systems (NeurIPS  2022)

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.05927v1) [paper-pdf](http://arxiv.org/pdf/2210.05927v1)

**Authors**: Yongyuan Liang, Yanchao Sun, Ruijie Zheng, Furong Huang

**Abstract**: Recent studies reveal that a well-trained deep reinforcement learning (RL) policy can be particularly vulnerable to adversarial perturbations on input observations. Therefore, it is crucial to train RL agents that are robust against any attacks with a bounded budget. Existing robust training methods in deep RL either treat correlated steps separately, ignoring the robustness of long-term rewards, or train the agents and RL-based attacker together, doubling the computational burden and sample complexity of the training process. In this work, we propose a strong and efficient robust training framework for RL, named Worst-case-aware Robust RL (WocaR-RL) that directly estimates and optimizes the worst-case reward of a policy under bounded l_p attacks without requiring extra samples for learning an attacker. Experiments on multiple environments show that WocaR-RL achieves state-of-the-art performance under various strong attacks, and obtains significantly higher training efficiency than prior state-of-the-art robust training methods. The code of this work is available at https://github.com/umd-huang-lab/WocaR-RL.



## **48. On the Limitations of Stochastic Pre-processing Defenses**

cs.LG

Accepted by Proceedings of the 36th Conference on Neural Information  Processing Systems

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2206.09491v3) [paper-pdf](http://arxiv.org/pdf/2206.09491v3)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz, Nicolas Papernot

**Abstract**: Defending against adversarial examples remains an open problem. A common belief is that randomness at inference increases the cost of finding adversarial inputs. An example of such a defense is to apply a random transformation to inputs prior to feeding them to the model. In this paper, we empirically and theoretically investigate such stochastic pre-processing defenses and demonstrate that they are flawed. First, we show that most stochastic defenses are weaker than previously thought; they lack sufficient randomness to withstand even standard attacks like projected gradient descent. This casts doubt on a long-held assumption that stochastic defenses invalidate attacks designed to evade deterministic defenses and force attackers to integrate the Expectation over Transformation (EOT) concept. Second, we show that stochastic defenses confront a trade-off between adversarial robustness and model invariance; they become less effective as the defended model acquires more invariance to their randomization. Future work will need to decouple these two effects. We also discuss implications and guidance for future research.



## **49. Adversarial Attack Against Image-Based Localization Neural Networks**

cs.CV

13 pages, 10 figures

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2210.06589v1) [paper-pdf](http://arxiv.org/pdf/2210.06589v1)

**Authors**: Meir Brand, Itay Naeh, Daniel Teitelman

**Abstract**: In this paper, we present a proof of concept for adversarially attacking the image-based localization module of an autonomous vehicle. This attack aims to cause the vehicle to perform a wrong navigational decisions and prevent it from reaching a desired predefined destination in a simulated urban environment. A database of rendered images allowed us to train a deep neural network that performs a localization task and implement, develop and assess the adversarial pattern. Our tests show that using this adversarial attack we can prevent the vehicle from turning at a given intersection. This is done by manipulating the vehicle's navigational module to falsely estimate its current position and thus fail to initialize the turning procedure until the vehicle misses the last opportunity to perform a safe turn in a given intersection.



## **50. Indicators of Attack Failure: Debugging and Improving Optimization of Adversarial Examples**

cs.LG

Accepted at NeurIPS 2022

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2106.09947v3) [paper-pdf](http://arxiv.org/pdf/2106.09947v3)

**Authors**: Maura Pintor, Luca Demetrio, Angelo Sotgiu, Ambra Demontis, Nicholas Carlini, Battista Biggio, Fabio Roli

**Abstract**: Evaluating robustness of machine-learning models to adversarial examples is a challenging problem. Many defenses have been shown to provide a false sense of robustness by causing gradient-based attacks to fail, and they have been broken under more rigorous evaluations. Although guidelines and best practices have been suggested to improve current adversarial robustness evaluations, the lack of automatic testing and debugging tools makes it difficult to apply these recommendations in a systematic manner. In this work, we overcome these limitations by: (i) categorizing attack failures based on how they affect the optimization of gradient-based attacks, while also unveiling two novel failures affecting many popular attack implementations and past evaluations; (ii) proposing six novel indicators of failure, to automatically detect the presence of such failures in the attack optimization process; and (iii) suggesting a systematic protocol to apply the corresponding fixes. Our extensive experimental analysis, involving more than 15 models in 3 distinct application domains, shows that our indicators of failure can be used to debug and improve current adversarial robustness evaluations, thereby providing a first concrete step towards automatizing and systematizing them. Our open-source code is available at: https://github.com/pralab/IndicatorsOfAttackFailure.



